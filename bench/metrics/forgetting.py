"""
PackWatch-Bench — Forgetting metrics (Part E stress test).

Measures whether Pack Watcher's continual-learning mechanism (CNL + RaPO)
prevents catastrophic forgetting across the sequential attack-family curriculum.

Core metric: Backward Transfer (BWT)
    BWT = (1/T-1) * Σ_{i=0}^{T-2} (R_{T,i} − R_{i,i})

    R_{j,i} = accuracy on family i after training through families 0..j
    R_{i,i} = accuracy on family i immediately after training it (best we ever saw)

    BWT < 0  → forgetting (accuracy dropped)
    BWT = 0  → no forgetting
    BWT > 0  → positive backward transfer (earlier families actually improved)

    Pass threshold: BWT > -0.10 (we allow at most 10pp accuracy drop on old families)

Also produces:
    final_accuracy_matrix: full [n_families × n_stages] heatmap data
    forgetting_per_family: per-family accuracy drop from peak to final
"""

from __future__ import annotations

import numpy as np

from packwatcher.types import ForgettingResult


def compute_forgetting_metrics(
    accuracy_matrix: list[list[float]],  # [n_stages][n_families] as returned by curriculum
    family_names:    list[str],
) -> ForgettingResult:
    """
    Compute forgetting benchmark from the accuracy matrix recorded during
    the sequential curriculum run.

    accuracy_matrix[stage][family_idx] = accuracy on family_idx after training
    through stages 0..stage.

    The diagonal (stage==family_idx) gives the accuracy immediately after
    each family was trained.

    Args:
        accuracy_matrix: [n_stages][n_families] float matrix, values in [0,1]
        family_names:    ordered list of family names

    Returns ForgettingResult.
    """
    mat   = np.array(accuracy_matrix, dtype=float)   # [S, F]
    S, F  = mat.shape

    assert S == F, (
        f"accuracy_matrix must be square (n_stages == n_families). "
        f"Got shape [{S}, {F}]."
    )

    # BWT: for each family i < final_stage, compare final acc to acc at peak
    deltas: list[float] = []
    for i in range(F - 1):
        acc_at_i    = mat[i, i]        # accuracy immediately after training family i
        acc_final_i = mat[-1, i]       # accuracy on family i after all training
        deltas.append(acc_final_i - acc_at_i)

    bwt = float(np.mean(deltas)) if deltas else 0.0

    return ForgettingResult(
        backward_transfer    = bwt,
        final_accuracy_matrix = mat.tolist(),
        family_names         = list(family_names),
    )


def forgetting_per_family(result: ForgettingResult) -> dict[str, float]:
    """
    Convenience: per-family accuracy drop from peak to final.
    Negative = forgetting.
    """
    mat = np.array(result.final_accuracy_matrix)
    F   = len(result.family_names)
    out: dict[str, float] = {}
    for i, name in enumerate(result.family_names[: F - 1]):
        acc_at    = mat[i, i]
        acc_final = mat[-1, i]
        out[name] = float(acc_final - acc_at)
    return out
