"""
Part E — Sequential attack-family curriculum manager.

Manages the ordered list of attack families used for continual-learning stress tests.
Provides:
    - Iteration over families in order (next_family())
    - Per-stage evaluation on ALL families (evaluate_all_families())
    - Forgetting-curve data: accuracy matrix [n_families × n_stages]

The forgetting test (blueprint Section 1, Part E):
    "run sequential curriculum — teach watcher attack-family 1, then 2, then 3...
     check watcher still catch family-1 pattern after learning family-5."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class FamilyDataset:
    """Training/eval data for one attack family."""
    family:     str
    X_train:    torch.Tensor   # [N_train, input_dim]
    y_train:    torch.Tensor   # [N_train]  binary labels
    X_eval:     torch.Tensor   # [N_eval,  input_dim]
    y_eval:     torch.Tensor   # [N_eval]


@dataclass
class CurriculumResult:
    """Accuracy matrix produced by running the full curriculum."""
    family_names:       list[str]
    # accuracy_matrix[stage][family] = accuracy after training through stages 0..stage
    accuracy_matrix:    list[list[float]]
    backward_transfer:  float   # BWT = mean(acc_after[i] - acc_at[i]) for i < current


class AttackFamilyCurriculum:
    """
    Manages the sequential training curriculum over attack families.

    Usage:
        curriculum = AttackFamilyCurriculum.from_data_dir("data/splits/curriculum/")
        for family_ds in curriculum:
            trainer.train_on(family_ds)
            accs = curriculum.evaluate_all(model, eval_fn)
    """

    def __init__(self, families: list[FamilyDataset]) -> None:
        self.families     = families
        self._stage_idx   = 0
        self._family_names = [f.family for f in families]
        # accuracy_matrix[stage][family_idx]
        self._accuracy_matrix: list[list[float]] = []

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self):
        self._stage_idx = 0
        return self

    def __next__(self) -> FamilyDataset:
        if self._stage_idx >= len(self.families):
            raise StopIteration
        ds = self.families[self._stage_idx]
        self._stage_idx += 1
        return ds

    def __len__(self) -> int:
        return len(self.families)

    @property
    def current_stage(self) -> int:
        return self._stage_idx

    @property
    def family_names(self) -> list[str]:
        return list(self._family_names)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(
        self,
        eval_fn: Callable[[torch.Tensor, torch.Tensor], float],
    ) -> list[float]:
        """
        Evaluate model on ALL families using eval_fn.

        eval_fn(X_eval, y_eval) -> accuracy (float in [0,1])

        Returns list of accuracies, one per family.
        Records stage in the internal accuracy matrix.
        """
        accs: list[float] = []
        for ds in self.families:
            acc = eval_fn(ds.X_eval, ds.y_eval)
            accs.append(float(acc))
        self._accuracy_matrix.append(accs)
        return accs

    def compute_backward_transfer(self) -> float:
        """
        Backward Transfer (BWT): measures how much the model forgets.

            BWT = (1 / T-1) * Σ_{i=0}^{T-2} (acc_final[i] - acc_at_i[i])

        acc_final[i]   = accuracy on family i AFTER training through ALL families
        acc_at_i[i]    = accuracy on family i IMMEDIATELY after training family i

        Negative BWT = forgetting.  0 = no forgetting.  Positive = positive transfer.
        """
        T = len(self.families)
        if len(self._accuracy_matrix) < T:
            raise RuntimeError(
                f"Need {T} evaluation stages to compute BWT "
                f"(have {len(self._accuracy_matrix)}). "
                "Call evaluate_all() once per curriculum stage."
            )
        # acc_at_i[i] = _accuracy_matrix[i][i]   (diagonal)
        # acc_final[i] = _accuracy_matrix[-1][i]  (last row)
        deltas: list[float] = []
        for i in range(T - 1):
            acc_at   = self._accuracy_matrix[i][i]
            acc_final = self._accuracy_matrix[-1][i]
            deltas.append(acc_final - acc_at)

        return float(np.mean(deltas)) if deltas else 0.0

    def build_result(self) -> CurriculumResult:
        bwt = self.compute_backward_transfer()
        return CurriculumResult(
            family_names=self._family_names,
            accuracy_matrix=self._accuracy_matrix,
            backward_transfer=bwt,
        )

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_tensors(
        cls,
        data: dict[str, dict[str, torch.Tensor]],
    ) -> "AttackFamilyCurriculum":
        """
        Build curriculum from a dict:
            {family_name: {"X_train": ..., "y_train": ..., "X_eval": ..., "y_eval": ...}}

        Families are ordered by their dict insertion order (Python 3.7+).
        """
        families: list[FamilyDataset] = []
        for name, tensors in data.items():
            families.append(FamilyDataset(
                family  = name,
                X_train = tensors["X_train"],
                y_train = tensors["y_train"],
                X_eval  = tensors["X_eval"],
                y_eval  = tensors["y_eval"],
            ))
        return cls(families)

    @classmethod
    def from_data_dir(cls, base_dir: str) -> "AttackFamilyCurriculum":
        """
        Load curriculum from directory structure:
            base_dir/
                family_order.json        — ordered list of family names
                {family_name}/
                    X_train.npy, y_train.npy, X_eval.npy, y_eval.npy
        """
        base = Path(base_dir)
        order_path = base / "family_order.json"
        if not order_path.exists():
            raise FileNotFoundError(f"family_order.json not found in {base_dir}")

        with open(order_path) as f:
            family_order: list[str] = json.load(f)

        families: list[FamilyDataset] = []
        for name in family_order:
            fdir = base / name
            families.append(FamilyDataset(
                family  = name,
                X_train = torch.from_numpy(np.load(str(fdir / "X_train.npy"))),
                y_train = torch.from_numpy(np.load(str(fdir / "y_train.npy"))),
                X_eval  = torch.from_numpy(np.load(str(fdir / "X_eval.npy"))),
                y_eval  = torch.from_numpy(np.load(str(fdir / "y_eval.npy"))),
            ))
        return cls(families)
