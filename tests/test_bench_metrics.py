"""
tests/test_bench_metrics.py — Unit tests for PackWatch-Bench metric modules.
"""

import pytest
import numpy as np
from typing import Optional

from bench.metrics.detection import compute_detection_metrics
from bench.metrics.prediction import compute_prediction_metrics
from bench.metrics.surgical import compute_surgical_metrics, ScenarioSurgicalRecord
from bench.metrics.forgetting import compute_forgetting_metrics, forgetting_per_family
from bench.metrics.false_positive import compute_false_positive_metrics
from bench.metrics.dual_use import compute_dual_use_metrics


# ============================================================
# Detection metrics
# ============================================================

class TestDetectionMetrics:

    def test_perfect_known_detection(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_prob = np.array([0.9, 0.95, 0.05, 0.1, 0.85, 0.2])
        result = compute_detection_metrics(y_true, y_prob, y_true, y_prob)
        assert result.f1_known > 0.9
        assert result.f1_novel > 0.9

    def test_random_known_detection(self):
        np.random.seed(0)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        result = compute_detection_metrics(y_true, y_prob, y_true, y_prob)
        assert 0.0 <= result.f1_known <= 1.0
        assert 0.0 <= result.f1_novel <= 1.0
        assert 0.0 <= result.generalization_gap

    def test_empty_novel_split(self):
        y_true = np.array([1, 0, 1])
        y_prob = np.array([0.8, 0.2, 0.9])
        result = compute_detection_metrics(y_true, y_prob,
                                           np.array([]), np.array([]))
        assert result.f1_novel == 0.0

    def test_generalization_gap_formula(self):
        y_true = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])
        # Perfect known
        y_novel_true = np.array([1, 0])
        y_novel_prob = np.array([0.6, 0.4])   # weaker novel
        result = compute_detection_metrics(y_true, y_prob, y_novel_true, y_novel_prob)
        expected_gap = result.f1_novel / result.f1_known if result.f1_known > 1e-6 else 0.0
        assert abs(result.generalization_gap - min(expected_gap, 2.0)) < 1e-4

    def test_all_ones_prediction(self):
        y_true = np.array([1, 1, 1])
        y_prob = np.ones(3)
        result = compute_detection_metrics(y_true, y_prob, y_true, y_prob)
        assert result.recall_known == 1.0

    def test_all_zeros_ground_truth(self):
        y_true = np.zeros(5)
        y_prob = np.array([0.8, 0.3, 0.7, 0.1, 0.9])
        result = compute_detection_metrics(y_true, y_prob, y_true, y_prob)
        # Recall on empty positive class should be 0
        assert result.f1_known == 0.0


# ============================================================
# Prediction metrics
# ============================================================

class TestPredictionMetrics:

    def test_basic_run(self):
        alert_histories = [[True, False, True, False, False]] * 3
        tip_timesteps   = [1, 2, 0]
        alert_probs     = [[0.8, 0.3, 0.9, 0.1, 0.2]] * 3
        result = compute_prediction_metrics(alert_histories, tip_timesteps, alert_probs)
        assert result.lead_time.n_events == 3
        assert 0.0 <= result.ece <= 1.0
        assert 0.0 <= result.fn_rate <= 1.0

    def test_no_tips(self):
        alert_histories = [[False] * 5] * 3
        tip_timesteps   = [None] * 3
        alert_probs     = [[0.1] * 5] * 3
        result = compute_prediction_metrics(alert_histories, tip_timesteps, alert_probs)
        assert result.lead_time.n_events == 0
        assert result.fn_rate == 0.0

    def test_all_missed_tips(self):
        alert_histories = [[False, False, False]] * 2
        tip_timesteps   = [1, 2]
        alert_probs     = [[0.1, 0.1, 0.1]] * 2
        result = compute_prediction_metrics(alert_histories, tip_timesteps, alert_probs)
        assert result.fn_rate == 1.0

    def test_no_missed_tips(self):
        alert_histories = [[True, False, False]] * 2
        tip_timesteps   = [1, 2]   # alerted at or before tip
        alert_probs     = [[0.9, 0.2, 0.1]] * 2
        result = compute_prediction_metrics(alert_histories, tip_timesteps, alert_probs)
        assert result.fn_rate == 0.0

    def test_ece_in_range(self):
        np.random.seed(42)
        n               = 100
        alert_histories = [[np.random.rand() > 0.5] * 5] * n
        tip_timesteps   = [np.random.randint(0, 5) if np.random.rand() > 0.3 else None
                           for _ in range(n)]
        alert_probs     = [[np.random.rand()] * 5 for _ in range(n)]
        result = compute_prediction_metrics(alert_histories, tip_timesteps, alert_probs)
        assert 0.0 <= result.ece <= 1.0


# ============================================================
# Surgical metrics
# ============================================================

class TestSurgicalMetrics:

    def _make_record(self, perf_freeze: float, perf_shutdown: float,
                     drift_stopped: bool = True) -> ScenarioSurgicalRecord:
        return ScenarioSurgicalRecord(
            scenario_id          = "test",
            perf_after_freeze    = perf_freeze,
            perf_after_shutdown  = perf_shutdown,
            phi_before           = 0.7,
            phi_after_freeze     = 0.4 if drift_stopped else 0.8,
            drift_stopped        = drift_stopped,
        )

    def test_empty_records(self):
        result = compute_surgical_metrics([])
        assert result.retained_perf_ratio == 0.0
        assert result.drift_stopped_rate  == 0.0

    def test_perfect_surgical(self):
        """Freeze preserves full performance; shutdown gives 0."""
        records = [self._make_record(1.0, 0.0)] * 5
        result  = compute_surgical_metrics(records)
        # ratio = 1.0 / 0.01 = 100 → capped at 5.0
        assert result.retained_perf_ratio > 1.0

    def test_drift_stopped_rate(self):
        records = [
            self._make_record(0.9, 0.5, drift_stopped=True),
            self._make_record(0.8, 0.5, drift_stopped=True),
            self._make_record(0.7, 0.5, drift_stopped=False),
        ]
        result = compute_surgical_metrics(records)
        assert abs(result.drift_stopped_rate - 2.0/3.0) < 1e-6

    def test_ratio_better_than_shutdown(self):
        records = [self._make_record(0.85, 0.0)] * 3
        result  = compute_surgical_metrics(records)
        assert result.retained_perf_ratio > 1.0

    def test_scenarios_recorded(self):
        records = [self._make_record(0.9, 0.5)] * 2
        result  = compute_surgical_metrics(records)
        assert len(result.scenarios) == 2


# ============================================================
# Forgetting metrics
# ============================================================

class TestForgettingMetrics:

    def test_no_forgetting(self):
        # All accuracies stay at 1.0
        mat = [[1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0],
               [1.0, 1.0, 1.0]]
        result = compute_forgetting_metrics(mat, ["a", "b", "c"])
        assert abs(result.backward_transfer - 0.0) < 1e-6

    def test_full_forgetting(self):
        # Learned perfectly then completely forgot
        mat = [[1.0, 0.5, 0.5],
               [0.0, 1.0, 0.5],
               [0.0, 0.0, 1.0]]
        result = compute_forgetting_metrics(mat, ["a", "b", "c"])
        assert result.backward_transfer < 0

    def test_positive_backward_transfer(self):
        # Old families improved after learning new ones
        mat = [[0.5, 0.5, 0.5],
               [0.7, 0.8, 0.5],
               [0.9, 0.9, 1.0]]
        result = compute_forgetting_metrics(mat, ["a", "b", "c"])
        assert result.backward_transfer > 0

    def test_non_square_raises(self):
        with pytest.raises(AssertionError):
            compute_forgetting_metrics([[1.0, 0.5]], ["a", "b"])

    def test_forgetting_per_family_dict(self):
        mat = [[1.0, 0.5, 0.5],
               [0.8, 1.0, 0.5],
               [0.6, 0.7, 1.0]]
        result = compute_forgetting_metrics(mat, ["a", "b", "c"])
        per_fam = forgetting_per_family(result)
        assert "a" in per_fam
        assert "b" in per_fam
        assert per_fam["a"] == pytest.approx(0.6 - 1.0, abs=1e-5)


# ============================================================
# False-positive metrics
# ============================================================

class TestFalsePositiveMetrics:

    def test_zero_fp(self):
        n = 20
        result = compute_false_positive_metrics(
            trigger_flags = [False] * n,
            tribe_sizes   = [3] * n,
            task_types    = ["coding"] * n,
            modes         = ["white_box"] * n,
            is_honest     = [True] * n,
        )
        assert result.fpr == 0.0

    def test_fifty_percent_fp(self):
        n = 10
        result = compute_false_positive_metrics(
            trigger_flags = [True, False] * 5,
            tribe_sizes   = [4] * n,
            task_types    = ["research"] * n,
            modes         = ["white_box"] * n,
            is_honest     = [True] * n,
        )
        assert abs(result.fpr - 0.5) < 1e-6

    def test_ignores_non_honest_steps(self):
        result = compute_false_positive_metrics(
            trigger_flags = [True, True, True],
            tribe_sizes   = [3, 3, 3],
            task_types    = ["coding"] * 3,
            modes         = ["white_box"] * 3,
            is_honest     = [False, False, True],  # only last is honest
        )
        assert result.fpr == 1.0   # 1 honest step, triggered once

    def test_breakdown_by_tribe_size(self):
        result = compute_false_positive_metrics(
            trigger_flags = [True, False, True, False],
            tribe_sizes   = [2, 2, 4, 4],
            task_types    = ["t"] * 4,
            modes         = ["white_box"] * 4,
            is_honest     = [True] * 4,
        )
        assert 2 in result.by_tribe_size
        assert 4 in result.by_tribe_size
        assert abs(result.by_tribe_size[2] - 0.5) < 1e-6
        assert abs(result.by_tribe_size[4] - 0.5) < 1e-6

    def test_breakdown_by_mode(self):
        result = compute_false_positive_metrics(
            trigger_flags = [True, False],
            tribe_sizes   = [3, 3],
            task_types    = ["t"] * 2,
            modes         = ["white_box", "black_box"],
            is_honest     = [True, True],
        )
        assert result.by_mode["white_box"] == 1.0
        assert result.by_mode["black_box"] == 0.0


# ============================================================
# Dual-use metrics
# ============================================================

class TestDualUseMetrics:

    def test_zero_gap(self):
        result = compute_dual_use_metrics([False] * 10, [False] * 10)
        assert result.evasion_gap == 0.0
        assert result.is_dual_use_risk is False

    def test_positive_gap_triggers_risk(self):
        # Scout helps: with scout 80% succeed, without 50%
        result = compute_dual_use_metrics(
            success_flags_without_scout = [True] * 5 + [False] * 5,   # 50%
            success_flags_with_scout    = [True] * 8 + [False] * 2,   # 80%
        )
        assert result.evasion_gap > 0.0
        assert result.is_dual_use_risk is True

    def test_negative_gap_not_risk(self):
        # Scout actually makes attacks less effective (good!)
        result = compute_dual_use_metrics(
            success_flags_without_scout = [True] * 7 + [False] * 3,   # 70%
            success_flags_with_scout    = [True] * 3 + [False] * 7,   # 30%
        )
        assert result.evasion_gap < 0.0
        assert result.is_dual_use_risk is False

    def test_empty_inputs(self):
        result = compute_dual_use_metrics([], [])
        assert result.evasion_gap == 0.0
        assert result.is_dual_use_risk is False

    def test_gap_equals_difference(self):
        no_scout  = [True] * 4 + [False] * 6   # 0.4
        with_scout = [True] * 6 + [False] * 4  # 0.6
        result = compute_dual_use_metrics(no_scout, with_scout)
        assert abs(result.evasion_gap - 0.2) < 1e-6
        assert abs(result.success_rate_without_scout - 0.4) < 1e-6
        assert abs(result.success_rate_with_scout    - 0.6) < 1e-6
