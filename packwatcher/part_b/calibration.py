"""
Part B — Calibration tools for the world model's probabilistic predictions.

compute_ece():       Expected Calibration Error (ECE) for binary alert predictions.
reliability_diagram(): Per-bin accuracy vs confidence data for plotting.
lead_time_accuracy():  For each actual tip event, how many turns before it did we alert?

These are used by bench/metrics/prediction.py and by run_phase2.py to report
whether the world-model's confidence estimates are trustworthy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from packwatcher.types import LeadTimeStats


# ---------------------------------------------------------------------------
# Calibration result types
# ---------------------------------------------------------------------------

@dataclass
class CalibrationResult:
    ece: float
    bin_confidences: list[float]   # mean predicted prob per bin
    bin_accuracies:  list[float]   # mean actual outcome per bin
    bin_counts:      list[int]


# ---------------------------------------------------------------------------
# ECE / reliability diagram
# ---------------------------------------------------------------------------

def compute_ece(
    predicted_probs: np.ndarray,   # [N] floats in [0, 1]
    actual_outcomes: np.ndarray,   # [N] binary {0, 1}
    n_bins: int = 15,
) -> CalibrationResult:
    """
    Expected Calibration Error.

    ECE = Σ_b (|B_b| / N) × |acc(B_b) − conf(B_b)|

    Args:
        predicted_probs: model's predicted P(bad | step)
        actual_outcomes: ground-truth binary label (1 = actually bad)
    """
    assert len(predicted_probs) == len(actual_outcomes), "arrays must be same length"
    predicted_probs = np.asarray(predicted_probs, dtype=float)
    actual_outcomes = np.asarray(actual_outcomes, dtype=float)
    N = len(predicted_probs)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_conf: list[float] = []
    bin_acc:  list[float] = []
    bin_cnt:  list[int]   = []
    ece = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include right edge on last bin
        if i < n_bins - 1:
            mask = (predicted_probs >= lo) & (predicted_probs < hi)
        else:
            mask = (predicted_probs >= lo) & (predicted_probs <= hi)

        cnt = int(mask.sum())
        bin_cnt.append(cnt)

        if cnt == 0:
            bin_conf.append(float((lo + hi) / 2.0))
            bin_acc.append(0.0)
            continue

        conf = float(predicted_probs[mask].mean())
        acc  = float(actual_outcomes[mask].mean())
        bin_conf.append(conf)
        bin_acc.append(acc)
        ece += (cnt / N) * abs(conf - acc)

    return CalibrationResult(
        ece=float(ece),
        bin_confidences=bin_conf,
        bin_accuracies=bin_acc,
        bin_counts=bin_cnt,
    )


# ---------------------------------------------------------------------------
# Lead-time accuracy
# ---------------------------------------------------------------------------

def lead_time_accuracy(
    alert_histories: list[list[bool]],    # [n_episodes] × [n_turns] alert flags
    tip_timesteps:   list[Optional[int]], # actual tip turn per episode; None = no tip
) -> LeadTimeStats:
    """
    For each episode that had a real tipping point, find when the watcher first alerted
    (at or before the tip) and compute the lead time = tip_t − first_alert_t.

    Episodes where the watcher never alerted before the tip contribute 0 lead time
    (missed events). Episodes with no tip are excluded from lead-time stats.

    Returns LeadTimeStats.
    """
    assert len(alert_histories) == len(tip_timesteps), "arrays must be same length"

    lead_times: list[float] = []

    for alerts, tip_t in zip(alert_histories, tip_timesteps):
        if tip_t is None:
            # Healthy episode — not relevant for lead-time stat
            continue

        # Find the earliest alert at or before the tip
        first_alert: Optional[int] = None
        for t, alerted in enumerate(alerts):
            if alerted and t <= tip_t:
                first_alert = t
                break

        if first_alert is not None:
            lead_times.append(float(tip_t - first_alert))
        else:
            # Watcher missed this tip; count as 0 lead time
            lead_times.append(0.0)

    if not lead_times:
        return LeadTimeStats(mean=0.0, std=0.0, median=0.0, p10=0.0, p90=0.0, n_events=0)

    arr = np.array(lead_times, dtype=float)
    return LeadTimeStats(
        mean   = float(arr.mean()),
        std    = float(arr.std()),
        median = float(np.median(arr)),
        p10    = float(np.percentile(arr, 10)),
        p90    = float(np.percentile(arr, 90)),
        n_events = len(lead_times),
    )
